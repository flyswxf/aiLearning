"""运行：python learning_labs/multimodal_model_layers/test_verify.py"""

import torch

from model_components import (
    LlavaStyleVisionProjector,
    PerceiverCrossAttention,
    TinyMultimodalDecoder,
)


def main() -> None:
    torch.manual_seed(7)
    batch, patches, vision_dim, language_dim = 2, 16, 32, 48
    vision = torch.randn(batch, patches, vision_dim, requires_grad=True)

    projector = LlavaStyleVisionProjector(vision_dim, language_dim)
    projected = projector(vision)
    assert projected.shape == (batch, patches, language_dim)

    mask = torch.zeros(batch, patches, dtype=torch.bool)
    mask[1, -4:] = True
    perceiver = PerceiverCrossAttention(language_dim, num_queries=4, num_heads=4)
    compressed = perceiver(projected, mask)
    assert compressed.shape == (batch, 4, language_dim)

    input_ids = torch.randint(0, 1000, (batch, 10))
    decoder = TinyMultimodalDecoder(vocab_size=1000, dim=language_dim)
    logits = decoder(input_ids, compressed)
    assert logits.shape == (batch, 14, 1000)

    logits.mean().backward()
    assert vision.grad is not None
    assert torch.isfinite(vision.grad).all()
    print("PASS: vision -> projector -> cross-attention -> decoder")
    print(f"projected={tuple(projected.shape)}, compressed={tuple(compressed.shape)}")
    print(f"logits={tuple(logits.shape)}, gradient_ok={vision.grad is not None}")


if __name__ == "__main__":
    main()
