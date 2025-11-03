import * as tf from "@xenova/transformers"
import * as cs from "./cslib.js"
import cf from "../assets/configs.json" with { type: "json" }

async function tagImage(imagePath: string, labels: string[]) {
  const classifier = await tf.pipeline('zero-shot-image-classification', 'Xenova/clip-vit-large-patch14-336', { // 1. MODEL CHOICE
      cache_dir: './models', // where to save models
      quantized: false,
    }
  )
  const results = await classifier(imagePath, labels)
  return results
}

// Use it
for (let i of cs.range(2))
  tagImage(`./assets/${i}.png`, cf.labels).then(results => console.log(results))