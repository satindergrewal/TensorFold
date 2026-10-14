"""Tower reference: synthetic 2-frame 448x448 image -> [256, 4096] features.
Mirrors the lane's glm.py encode() math exactly, from raw checkpoint weights."""
import json, struct, glob, math
import torch
import torch.nn.functional as F

D = "/model"
# collect shard headers
def tensor(k):
    for f in sorted(glob.glob(D + "/model-*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        if k in hdr:
            v = hdr[k]
            a, b = v["data_offsets"]
            with open(f, "rb") as fh:
                fh.seek(8 + n + a)
                raw = fh.read(b - a)
            import numpy as np
            if v["dtype"] == "BF16":
                arr = np.frombuffer(raw, dtype="<u2").astype(np.uint16)
                t = torch.from_numpy(arr.copy()).view(torch.bfloat16)
            else:
                t = torch.frombuffer(bytearray(raw), dtype=getattr(torch, v["dtype"].lower()))
            return t.reshape(v["shape"]).float().cuda()
    raise KeyError(k)

cfg = json.load(open(D + "/config.json"))["vision_config"]
P, M, HID, DEPTH, HEADS = cfg["patch_size"], cfg["spatial_merge_size"], cfg["hidden_size"], cfg["depth"], cfg["num_heads"]
HD = HID // HEADS
OUT = cfg["out_hidden_size"]
LIMIT = cfg["swiglu_limit"]
EPS = cfg["rms_norm_eps"]
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda().view(3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda().view(3, 1, 1)
THETA = float((cfg.get("rope_parameters") or {}).get("rope_theta", 10000.0))

def rms(x, w):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS))

torch.manual_seed(7)
frames = (torch.rand(2, 3, 448, 448, device="cuda") * 255).to(torch.uint8)  # T=2 frames
T, C, H, W = frames.shape
gh, gw = H // P, W // P
x = (frames.float() - MEAN) / STD
x = x.view(T // 2, 2, C, gh // M, M, P, gw // M, M, P).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
x = x.reshape(T // 2 * gh * gw, C * 2 * P * P)  # [1024, 1176] patches

# tower
x = F.conv3d(x.view(-1, 3, 2, P, P).to(torch.bfloat16).float(),
             tensor("model.visual.patch_embed.proj.weight"), tensor("model.visual.patch_embed.proj.bias"),
             stride=(2, P, P))
S = gh * gw
x = x.view(1, S, HID)
spatial = HD // 2
inv = 1.0 / (THETA ** (torch.arange(0, spatial, 2, device="cuda") / spatial))
# NOTE: the lane's inv_freq: torch.arange(0, dim//4, 0.25)/(dim//4) -> [dim] via 4x repeat; build exactly:
dim = HD
inv_freq = inv  # [spatial/2] = 16; pos x2 -> 32; cat -> 64 = head_dim
hpos, wpos = torch.meshgrid(torch.arange(gh, device="cuda"), torch.arange(gw, device="cuda"), indexing="ij")
shape = (gh // M, M, gw // M, M)
pos = torch.stack([hpos.reshape(shape).transpose(1, 2).flatten(), wpos.reshape(shape).transpose(1, 2).flatten()], -1).float()
freqs = (pos[..., None] * inv_freq).flatten(1)
freqs = torch.cat([freqs, freqs], -1)
cos, sin = freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]

def rotate(x):
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin

for i in range(DEPTH):
    g = f"model.visual.blocks.{i}."
    h = rms(x, tensor(g + "norm1.weight"))
    qkv = F.linear(h, tensor(g + "attn.qkv.weight"), tensor(g + "attn.qkv.bias"))
    q, k, v = qkv.view(1, S, 3, HEADS, HD).unbind(2)
    q = rotate(rms(q, tensor(g + "attn.q_norm.weight")))
    k = rotate(rms(k, tensor(g + "attn.k_norm.weight")))
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=HD ** -0.5)
    x = x + F.linear(o.transpose(1, 2).reshape(1, S, HID), tensor(g + "attn.proj.weight"), tensor(g + "attn.proj.bias"))
    h = rms(x, tensor(g + "norm2.weight"))
    gate = F.linear(h, tensor(g + "mlp.gate_proj.weight"), tensor(g + "mlp.gate_proj.bias")).clamp(max=LIMIT)
    up = F.linear(h, tensor(g + "mlp.up_proj.weight"), tensor(g + "mlp.up_proj.bias")).clamp(-LIMIT, LIMIT)
    x = x + F.linear(F.silu(gate) * up, tensor(g + "mlp.down_proj.weight"), tensor(g + "mlp.down_proj.bias"))

x = rms(x, tensor("model.visual.post_layernorm.weight")).reshape(-1, M, M, HID)
x = F.conv2d(x.permute(0, 3, 1, 2), tensor("model.visual.downsample.weight"), tensor("model.visual.downsample.bias"), stride=M).view(-1, OUT)
x = F.linear(x, tensor("model.visual.merger.proj.weight"))
x = F.gelu(F.layer_norm(x, (OUT,), tensor("model.visual.merger.post_projection_norm.weight"), tensor("model.visual.merger.post_projection_norm.bias")))
gate = F.linear(x, tensor("model.visual.merger.gate_proj.weight")).clamp(max=LIMIT)
up = F.linear(x, tensor("model.visual.merger.up_proj.weight")).clamp(-LIMIT, LIMIT)
out = F.linear(F.silu(gate) * up, tensor("model.visual.merger.down_proj.weight"))

print("features:", tuple(out.shape), out.dtype)
print("row0 first8:", out[0, :8].tolist())
print("sum:", out.float().sum().item(), "sq:", out.float().square().sum().item())
out.to(torch.bfloat16).cpu().view(torch.uint16).numpy().tofile("/tmp/vision_ref_feat.bin")
frames.cpu().numpy().tofile("/tmp/vision_ref_frames.bin")
print("saved /tmp/vision_ref_feat.bin + frames")
