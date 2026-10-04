// One rank's side of the CUDA IPC one-shot all-gather (ipc.cu): the exported region, the peers' mapped regions,
// the local control words (epoch, arrival count, poison, last copy-engine launch), the mapped host error record,
// the copy-engine protocol's side streams, and the launches.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "ipc.h"

namespace {

void cuda_ok(cudaError_t code, const char* what) {
    if (code != cudaSuccess) {
        cudaGetLastError();                     // an API error is not sticky: clear it before raising
        TORCH_CHECK(false, "CUDA IPC all-gather: ", what, ": ", cudaGetErrorString(code));
    }
}

}  // namespace

// The launch geometry from the byte count alone, so every rank's launch of a gather matches: one block for each
// block_bytes of the shard (at least one, at most max_grid), the shard's 16-byte packs split into equal contiguous
// runs, and no empty block.
std::vector<int64_t> plan(int64_t nbytes, int64_t block_bytes, int64_t max_grid) {
    TORCH_CHECK(nbytes > 0 && block_bytes > 0 && max_grid > 0, "CUDA IPC all-gather: plan needs positive sizes");
    const int64_t packs = (nbytes + PACK_BYTES - 1) / PACK_BYTES;
    int64_t grid = (nbytes + block_bytes - 1) / block_bytes;
    grid = std::max<int64_t>(1, std::min<int64_t>({grid, max_grid, (int64_t)MAX_BLOCKS, packs}));
    const int64_t chunk = (packs + grid - 1) / grid;
    grid = (packs + chunk - 1) / chunk;
    return {grid, chunk};
}

class Gather {
  public:
    // slot_bytes: the flag protocol's slot (the largest gather); ll_bytes / ce_bytes: the largest gather the LL and
    // copy-engine protocols take (0: that protocol is off and has no slots); bands: (protocol, largest bytes) pairs
    // in rising order, the protocol a gather takes when a launch leaves the choice to its size
    Gather(int64_t world, int64_t rank, int64_t slot_bytes, int64_t ll_bytes, int64_t ce_bytes,
           std::vector<int64_t> bands, int64_t timeout_ns, int64_t block_bytes, int64_t threads)
        : world_(world), rank_(rank), slot_bytes_(slot_bytes), ll_bytes_(ll_bytes), ce_bytes_(ce_bytes),
          bands_(std::move(bands)), timeout_ns_(timeout_ns), block_bytes_(block_bytes), threads_(threads) {
        TORCH_CHECK(2 <= world && world <= MAX_WORLD, "CUDA IPC all-gather: 2 to ", MAX_WORLD, " ranks, not ", world);
        TORCH_CHECK(0 <= rank && rank < world, "CUDA IPC all-gather: rank ", rank, " of ", world);
        TORCH_CHECK(slot_bytes > 0 && slot_bytes % 256 == 0, "CUDA IPC all-gather: slot bytes a multiple of 256");
        TORCH_CHECK(0 <= ll_bytes && ll_bytes <= slot_bytes && 0 <= ce_bytes && ce_bytes <= slot_bytes,
                    "CUDA IPC all-gather: LL and copy-engine bytes 0 to the slot bytes");
        TORCH_CHECK(block_bytes >= PACK_BYTES, "CUDA IPC all-gather: block bytes at least ", PACK_BYTES);
        check_threads(threads);
        TORCH_CHECK(timeout_ns > 0, "CUDA IPC all-gather: a positive wait limit");
        TORCH_CHECK(bands_.size() % 2 == 0, "CUDA IPC all-gather: bands are (protocol, bytes) pairs");
        for (size_t i = 0; i < bands_.size(); i += 2) {
            const int64_t p = bands_[i], most = bands_[i + 1];
            TORCH_CHECK(p == PROTO_FLAG || (p == PROTO_LL && most <= ll_bytes) || (p == PROTO_CE && most <= ce_bytes),
                        "CUDA IPC all-gather: a band's protocol has no room for its size");
            TORCH_CHECK(most > 0 && most <= slot_bytes && (i == 0 || most > bands_[i - 1]),
                        "CUDA IPC all-gather: band sizes rise up to the slot bytes");
        }
        cuda_ok(cudaGetDevice(&device_), "cudaGetDevice");
        // Every block of a launch must be resident at once: a block that waits for its peers spins on an SM, and the
        // peers wait for all of this rank's blocks. Half the SMs (one block of up to 1,024 threads fits an SM) leaves
        // room for the kernels that run beside a gather; every rank must get the same value (compared at setup).
        int sms = 0;
        cuda_ok(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_), "cudaDeviceGetAttribute");
        max_grid_ = std::max<int64_t>(1, std::min<int64_t>(MAX_BLOCKS, sms / 2));
        layout_.slot_bytes = slot_bytes;
        layout_.ll_slot_bytes = ll_bytes > 0 ? 2 * ((ll_bytes + 255) / 256 * 256) : 0;   // 8-byte words of 4 bytes
        layout_.ce_slot_bytes = ce_bytes > 0 ? (ce_bytes + 255) / 256 * 256 : 0;
        layout_.ll_base = DATA_OFFSET + 2LL * (world - 1) * slot_bytes;
        layout_.ce_base = layout_.ll_base + 2LL * (world - 1) * layout_.ll_slot_bytes;
        region_bytes_ = layout_.ce_base + (world - 1) * layout_.ce_slot_bytes;
        std::memset(&regions_, 0, sizeof(regions_));
        cuda_ok(cudaMalloc(&region_, (size_t)region_bytes_), "cudaMalloc (the IPC region)");
        cuda_ok(cudaMalloc(&ctl_, 256), "cudaMalloc (control words)");
        cuda_ok(cudaHostAlloc(reinterpret_cast<void**>(&err_host_), 64, cudaHostAllocMapped | cudaHostAllocPortable),
                "cudaHostAlloc (error record)");
        std::memset(err_host_, 0, 64);
        long long* err_dev = nullptr;
        cuda_ok(cudaHostGetDevicePointer(reinterpret_cast<void**>(&err_dev), err_host_, 0), "cudaHostGetDevicePointer");
        cuda_ok(cudaMemset(region_, 0, (size_t)region_bytes_), "cudaMemset (the IPC region)");
        cuda_ok(cudaMemset(ctl_, 0, 256), "cudaMemset (control words)");
        cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize");   // zeroed before any peer can write a flag
        regions_.base[rank] = static_cast<char*>(region_);
        auto* c = static_cast<char*>(ctl_);
        ctl_words_.epoch = reinterpret_cast<unsigned long long*>(c);
        ctl_words_.arrive = reinterpret_cast<unsigned*>(c + 8);
        ctl_words_.poison = reinterpret_cast<unsigned*>(c + 16);
        ctl_words_.last_ce = reinterpret_cast<unsigned long long*>(c + 24);
        ctl_words_.stage = reinterpret_cast<unsigned*>(c + 32);
        ctl_words_.err = err_dev;
    }

    ~Gather() { close(); }

    // this rank's region as a CUDA IPC handle (CPU uint8 tensor)
    at::Tensor handle() const {
        TORCH_CHECK(region_ != nullptr, "CUDA IPC all-gather: closed");
        const c10::cuda::CUDAGuard guard(device_);
        cudaIpcMemHandle_t h;
        cuda_ok(cudaIpcGetMemHandle(&h, region_), "cudaIpcGetMemHandle");
        at::Tensor out = at::empty({(int64_t)sizeof(h)}, at::TensorOptions().dtype(at::kByte));
        std::memcpy(out.data_ptr(), &h, sizeof(h));
        return out;
    }

    // every rank's handle, in rank order (this rank's own is skipped): map the peers' regions; the copy-engine
    // protocol's side streams and events
    void connect(const std::vector<at::Tensor>& handles) {
        TORCH_CHECK((int64_t)handles.size() == world_, "CUDA IPC all-gather: one handle a rank");
        TORCH_CHECK(opened_.empty(), "CUDA IPC all-gather: already connected");
        const c10::cuda::CUDAGuard guard(device_);
        for (int64_t r = 0; r < world_; ++r) {
            if (r == rank_) continue;
            const at::Tensor h = handles[r].contiguous().to(at::kCPU);
            TORCH_CHECK(h.numel() * h.element_size() == (int64_t)sizeof(cudaIpcMemHandle_t),
                        "CUDA IPC all-gather: a handle is ", sizeof(cudaIpcMemHandle_t), " bytes");
            cudaIpcMemHandle_t handle;
            std::memcpy(&handle, h.data_ptr(), sizeof(handle));
            void* p = nullptr;
            const cudaError_t code = cudaIpcOpenMemHandle(&p, handle, cudaIpcMemLazyEnablePeerAccess);
            if (code != cudaSuccess) {
                cudaGetLastError();
                unmap();
                TORCH_CHECK(false, "CUDA IPC all-gather: cudaIpcOpenMemHandle (rank ", r, "'s region): ",
                            cudaGetErrorString(code));
            }
            opened_.push_back(p);
            regions_.base[r] = static_cast<char*>(p);
        }
        if (ce_bytes_ > 0) {
            cuda_ok(cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming), "cudaEventCreate");
            for (int64_t k = 1; k < world_; ++k) {
                cudaStream_t s = nullptr;
                cudaEvent_t e = nullptr;
                cuda_ok(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate");
                side_.push_back(s);
                cuda_ok(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "cudaEventCreate");
                join_.push_back(e);
            }
        }
    }

    // the protocol a gather of nbytes takes when the launch leaves the choice to its size
    int64_t choose(int64_t nbytes) const {
        for (size_t i = 0; i < bands_.size(); i += 2)
            if (nbytes <= bands_[i + 1]) return bands_[i];
        return PROTO_FLAG;
    }

    // recv [world * n] <- every rank's send [n] in rank order; contiguous CUDA tensors on this device, n bytes at
    // most slot_bytes. block_bytes / threads: the defaults when negative; protocol: -1 by the size (the bands), or
    // PROTO_FLAG / PROTO_LL / PROTO_CE (within its size). Every rank's matching call must pass the same block_bytes
    // and protocol (the defaults and bands are compared at setup).
    void run(const at::Tensor& send, at::Tensor recv, int64_t block_bytes, int64_t threads, int64_t protocol) {
        TORCH_CHECK(!opened_.empty(), "CUDA IPC all-gather: not connected");
        TORCH_CHECK(send.is_cuda() && recv.is_cuda() && send.is_contiguous() && recv.is_contiguous(),
                    "CUDA IPC all-gather: contiguous CUDA tensors");
        TORCH_CHECK(send.get_device() == device_ && recv.get_device() == device_,
                    "CUDA IPC all-gather: tensors on the region's device");
        const int64_t nbytes = send.numel() * (int64_t)send.element_size();
        TORCH_CHECK(nbytes > 0 && nbytes <= slot_bytes_, "CUDA IPC all-gather: 1 to ", slot_bytes_,
                    " bytes a rank, not ", nbytes);
        TORCH_CHECK(recv.numel() * (int64_t)recv.element_size() == world_ * nbytes,
                    "CUDA IPC all-gather: recv must hold world x send bytes");
        const int64_t t = threads < 0 ? threads_ : threads;
        check_threads(t);
        const int64_t proto = protocol < 0 ? choose(nbytes) : protocol;
        TORCH_CHECK(proto == PROTO_FLAG || (proto == PROTO_LL && nbytes <= ll_bytes_) ||
                        (proto == PROTO_CE && nbytes <= ce_bytes_),
                    "CUDA IPC all-gather: protocol ", proto, " has no room for ", nbytes, " bytes a rank");
        const std::vector<int64_t> geometry = plan(nbytes, block_bytes < 0 ? block_bytes_ : block_bytes, max_grid_);
        const int grid = (int)geometry[0];
        const c10::cuda::CUDAGuard guard(device_);
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        if (proto != PROTO_CE) {
            launch_gather(stream, (int)proto, send, recv, nbytes, geometry[1], grid, (int)t, regions_, layout_,
                          ctl_words_, timeout_ns_, (int)world_, (int)rank_);
            return;
        }
        // copy engine: wait for the peers' acknowledgements, one peer copy a peer on its own side stream (forked
        // from and joined back to the current stream, so a graph holds parallel copy branches), then the rest
        launch_ce_begin(stream, regions_, ctl_words_, timeout_ns_, (int)world_, (int)rank_);
        cuda_ok(cudaEventRecord(fork_, stream), "cudaEventRecord");
        for (int64_t k = 1; k < world_; ++k) {
            const int64_t p = (rank_ + k) % world_;
            const int64_t index = rank_ < p ? rank_ : rank_ - 1;
            char* dst = regions_.base[p] + layout_.ce_base + index * layout_.ce_slot_bytes;
            const cudaStream_t side = side_[k - 1];
            cuda_ok(cudaStreamWaitEvent(side, fork_, 0), "cudaStreamWaitEvent");
            cuda_ok(cudaMemcpyAsync(dst, send.data_ptr(), (size_t)nbytes, cudaMemcpyDefault, side),
                    "cudaMemcpyAsync (a peer copy)");
            cuda_ok(cudaEventRecord(join_[k - 1], side), "cudaEventRecord");
            cuda_ok(cudaStreamWaitEvent(stream, join_[k - 1], 0), "cudaStreamWaitEvent");
        }
        launch_ce_finish(stream, send, recv, nbytes, geometry[1], grid, (int)t, regions_, layout_, ctl_words_,
                         timeout_ns_, (int)world_, (int)rank_);
    }

    // the mapped host error record's address: {failed sequence (0: none), peer, block, waited ns}
    int64_t error_address() const { return reinterpret_cast<int64_t>(err_host_); }

    std::vector<int64_t> error() const {
        if (err_host_ == nullptr) return {0, 0, 0, 0};
        const volatile long long* e = err_host_;
        return {e[0], e[1], e[2], e[3]};
    }

    int64_t region_bytes() const { return region_bytes_; }

    // the most blocks a launch uses on this device (half its SMs, at most MAX_BLOCKS)
    int64_t max_grid() const { return max_grid_; }

    // the wait limit of later launches (a captured graph keeps the value it was captured with)
    void set_timeout(int64_t timeout_ns) {
        TORCH_CHECK(timeout_ns > 0, "CUDA IPC all-gather: a positive wait limit");
        timeout_ns_ = timeout_ns;
    }

    void close() {
        if (region_ == nullptr && opened_.empty() && ctl_ == nullptr && err_host_ == nullptr && side_.empty()) return;
        cudaSetDevice(device_);
        for (cudaStream_t s : side_) cudaStreamDestroy(s);
        for (cudaEvent_t e : join_) cudaEventDestroy(e);
        if (fork_ != nullptr) cudaEventDestroy(fork_);
        side_.clear();
        join_.clear();
        fork_ = nullptr;
        unmap();
        if (region_ != nullptr) cudaFree(region_);
        if (ctl_ != nullptr) cudaFree(ctl_);
        if (err_host_ != nullptr) cudaFreeHost(err_host_);
        region_ = ctl_ = nullptr;
        err_host_ = nullptr;
        cudaGetLastError();                     // teardown at exit: nothing to report
    }

  private:
    static void check_threads(int64_t threads) {
        TORCH_CHECK(threads >= 32 && threads <= MAX_THREADS && threads % 32 == 0,
                    "CUDA IPC all-gather: threads a block a multiple of 32 up to ", MAX_THREADS);
    }

    void unmap() {
        for (void* p : opened_) cudaIpcCloseMemHandle(p);
        opened_.clear();
        for (int64_t r = 0; r < world_; ++r)
            if (r != rank_) regions_.base[r] = nullptr;
    }

    int64_t world_, rank_, slot_bytes_, ll_bytes_, ce_bytes_;
    std::vector<int64_t> bands_;
    int64_t timeout_ns_, block_bytes_, threads_;
    int device_ = 0;
    int64_t max_grid_ = MAX_BLOCKS;
    int64_t region_bytes_ = 0;
    void* region_ = nullptr;
    void* ctl_ = nullptr;
    long long* err_host_ = nullptr;
    std::vector<void*> opened_;
    Regions regions_;
    Layout layout_{};
    Control ctl_words_{};
    cudaEvent_t fork_ = nullptr;
    std::vector<cudaStream_t> side_;
    std::vector<cudaEvent_t> join_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("plan", &plan, "(grid, packs a block) of a gather of nbytes a rank", py::arg("nbytes"),
          py::arg("block_bytes"), py::arg("max_grid") = (int64_t)MAX_BLOCKS);
    m.def("layout", []() {
        return std::vector<int64_t>{IPC_ABI, MAX_WORLD, MAX_BLOCKS, MAX_THREADS, PACK_BYTES, CE_CTL_OFFSET, DATA_OFFSET,
                                    (int64_t)sizeof(cudaIpcMemHandle_t), PROTO_FLAG, PROTO_LL, PROTO_CE};
    });
    py::class_<Gather>(m, "Gather")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, std::vector<int64_t>, int64_t, int64_t, int64_t>(),
             py::arg("world"), py::arg("rank"), py::arg("slot_bytes"), py::arg("ll_bytes"), py::arg("ce_bytes"),
             py::arg("bands"), py::arg("timeout_ns"), py::arg("block_bytes"), py::arg("threads"))
        .def("handle", &Gather::handle)
        .def("connect", &Gather::connect)
        .def("choose", &Gather::choose)
        .def("run", &Gather::run, py::arg("send"), py::arg("recv"), py::arg("block_bytes") = -1,
             py::arg("threads") = -1, py::arg("protocol") = -1)
        .def("error_address", &Gather::error_address)
        .def("error", &Gather::error)
        .def("region_bytes", &Gather::region_bytes)
        .def("max_grid", &Gather::max_grid)
        .def("set_timeout", &Gather::set_timeout)
        .def("close", &Gather::close);
}
