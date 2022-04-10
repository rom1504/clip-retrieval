/* globals fetch */

import JsonBigint from 'json-bigint'

export default class ClipService {
  constructor (backend) {
    this.backend = backend
  }

  async getIndices () {
    const result = JsonBigint.parse(await (await fetch(this.backend + `/indices-list`, {
    })).text())

    return result
  }

  async callClipService (text, image, imageUrl, modality, numImages, indexName, numResultIds, useMclip, hideDuplicateImages, useSafetyModel, useViolenceDetector) {
    console.log('calling', text, numImages)
    const result = JsonBigint.parse(await (await fetch(this.backend + `/knn-service`, {
      method: 'POST',
      body: JSON.stringify({
        text,
        image,
        'image_url': imageUrl,
        modality,
        'num_images': numImages,
        'indice_name': indexName,
        'num_result_ids': numResultIds,
        'use_mclip': useMclip,
        'deduplicate': hideDuplicateImages,
        'use_safety_model': useSafetyModel,
        'use_violence_detector': useViolenceDetector
      })
    })).text())

    return result
  }

  async getMetadata (ids, indexName) {
    const result = JsonBigint.parse(await (await fetch(this.backend + `/metadata`, {
      method: 'POST',
      body: JSON.stringify({
        ids,
        'indice_name': indexName
      })
    })).text())

    return result
  }
}
