/* globals fetch */

import JsonBigint from 'json-bigint'

/*
const results = await this.clipService.callClipService(this.text, null, "image", this.numKnnImages, this.currentIndex)
*/

export default class ClipService {
  constructor (backend) {
    this.backend = backend
  }

  async getIndices () {
    const result = JsonBigint.parse(await (await fetch(this.backend + `/indices-list`, {
    })).text())

    return result
  }

  async callClipService (text, image, modality, numImages, indexName) {
    console.log('calling', text, numImages)
    const result = JsonBigint.parse(await (await fetch(this.backend + `/knn-service`, {
      method: 'POST',
      body:JSON.stringify({
        text,
        image,
        modality,
        'num_images': numImages,
        'indice_name': indexName
      })
    })).text())

    return result
  }
}
