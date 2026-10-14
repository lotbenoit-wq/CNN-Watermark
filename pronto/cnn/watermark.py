import torch


class WatermarkEmbedder:
    """
    Framework de Uchida et al. - méthode X_random.
    Embeds un T-bit watermark dans les poids d'une couche conv.
    """

    def __init__(self, target_layer, T=256, secret_key=42, device='cpu'):
        self.target_layer = target_layer
        self.T = T
        self.secret_key = secret_key
        self.device = device

        weight_shape = target_layer.weight.shape
        self.L = weight_shape[0]
        self.D = weight_shape[1]
        self.S = weight_shape[2]
        self.M = self.S * self.S * self.D

        assert self.M >= T, (
            f"M={self.M} doit être >= T={T}. "
            f"Choisis une couche avec plus de paramètres ou réduis T."
        )

        self.b = torch.ones(T, device=device)

        rng = torch.Generator(device='cpu')
        rng.manual_seed(secret_key)
        self.X = torch.randn(T, self.M, generator=rng).to(device)

        print(f"[Watermark] Layer shape: {list(weight_shape)}, M={self.M}, T={T}")

    def _get_w(self):
        W = self.target_layer.weight
        w = W.mean(dim=0).flatten()
        return w

    def compute_embedding_loss(self):
        w = self._get_w()
        z = torch.matmul(self.X, w)
        y = torch.sigmoid(z)
        loss = -torch.sum(
            self.b * torch.log(y + 1e-8) + (1 - self.b) * torch.log(1 - y + 1e-8)
        )  #On aurait pu utiliser la librairie
        return loss

    def extract_watermark(self):
        with torch.no_grad():
            w = self._get_w()
            z = torch.matmul(self.X, w)
            raw = torch.sigmoid(z)
            bits = (z >= 0).float()
        return bits, raw

    def bit_error_rate(self):
        bits, _ = self.extract_watermark()
        return (bits != self.b).float().mean().item()


def save_secret_key(watermark_embedder, path, model_name):
    import os
    import torch
    os.makedirs(path, exist_ok=True)
    secret_filepath = os.path.join(path, f"{model_name}_SECRET_KEY.pth")
    secret_dict = {
        'secret_key': watermark_embedder.secret_key,
        'T': watermark_embedder.T,
        'target_layer_name': 'conv3',
        'watermark_bits': watermark_embedder.b.cpu(),
        'X_matrix': watermark_embedder.X.cpu(),
    }
    torch.save(secret_dict, secret_filepath)
    print(f"  Clé secrète sauvegardée: {secret_filepath}")
    print(f"  NE PAS PARTAGER CE FICHIER!")
    return secret_filepath


def load_watermark_from_key(secret_path, model, device='cpu'):
    secret = torch.load(secret_path, map_location=device)
    target_layer = getattr(model, secret['target_layer_name'])
    watermark = WatermarkEmbedder(
        target_layer=target_layer,
        T=secret['T'],
        secret_key=secret['secret_key'],
        device=device
    )
    return watermark
