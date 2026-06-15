# Configuration & Settings

Deep Pavements Lite can be customized by modifying environment variables, parameters files, or the package constants.

---

## ⚙️ Configuration Files

### `params.json`
Located at the root of the repository, this file stores basic runtime parameters:
```json
{
  "max_images": null,
  "debug": false,
  "half_res": false,
  "quarter_res": false,
  "workers": 1
}
```

---

## 📌 Constant Definitions (`modules.constants`)

Core settings are centralized in `modules/constants.py`:

- **`DEVICE`**: Automatically set to `"cuda"` if a GPU is available; otherwise defaults to `"cpu"`.
- **`CLIP_ARCHITECTURE`**: Set to `"ViT-B/32"`.
- **`pathway_categories`**: Defines target segmentation classes: `["roads", "sidewalks", "car"]`.
- **`PATHWAY_CLASS_MAPPING`**: Maps target classes to Cityscapes IDs:
  - `roads`: Class ID `0`
  - `sidewalks`: Class ID `1`
  - `car`: Class ID `13`
- **`default_surfaces`**: The 11 surface types classified by CLIP:
  - `asphalt`, `concrete`, `concrete_plates`, `grass`, `ground`, `sett`, `paving_stones`, `cobblestone`, `gravel`, `sand`, `compacted`.

---

## 🌐 Environment Variables

Use environment variables to customize runtime behavior:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `MAPILLARY_API` | Your Mapillary developer access token. | None |
| `TO_PRECACHE` | If `true`, downloads models at build-time (used in Dockerfile). | `true` |

---

## 🤖 Model Customization

By default, the pipeline automatically fetches the fine-tuned CLIP model from Hugging Face:
- **Repository:** `kauevestena/clip-vit-base-patch32-finetuned-surface-materials`
- **Filename:** `pytorch_model.bin`
- **Direct Download URL:** `https://huggingface.co/kauevestena/clip-vit-base-patch32-finetuned-surface-materials/resolve/main/model.pt`

To use a custom model, place your model weight file at the root named `deep_pavements_clip_model.pt`.
