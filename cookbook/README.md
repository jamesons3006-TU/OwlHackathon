# Rice Bowl Cookbook

A single-page cookbook of Asian home dishes (Thai, Japanese, Filipino, Korean, Vietnamese, Chinese, Indian) with a personal kitchen log.

Open `index.html` in a browser. There is no build step and no server.

- **Recipes**: filter by cuisine, search by dish or ingredient, cap calories, sort, and filter for vegetarian or mild dishes. Each recipe has a servings scaler, an ingredient checklist, tap-to-complete steps, and a Nutrition Facts panel (per serving or for the whole pot).
- **My Kitchen**: log what you cooked with a date, notes, a taste rating and a photo. Each photo gets a score out of 10 with a breakdown and two tips.

Storage and rating depend on where the page runs:

| Where | Entries and photos | Photo rating |
|---|---|---|
| Opened as a local file | IndexedDB in that browser | On-device check of lighting, contrast, color, sharpness and framing |
| Published as a Claude artifact | The artifact's database and asset store | Claude looks at the photo and scores presentation, color, execution, authenticity and photo quality |

Nutrition values are per-serving estimates from typical ingredient values.
