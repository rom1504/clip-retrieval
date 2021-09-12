/* globals customElements, FileReader */
import { LitElement, html, css } from 'lit-element'
import ClipService from './clip-service'

class ClipFront extends LitElement {
  constructor () {
    super()
    const urlParams = new URLSearchParams(window.location.search)
    const back = urlParams.get('back')
    const model = urlParams.get('model')
    const query = urlParams.get('query')
    if (model != null) {
      this.currentIndex = model
    } else {
      this.currentIndex = ''
    }
    if (back != null) {
      this.backendHost = back
    } else {
      this.backendHost = 'https://clip.rom1504.fr' // put something here
    }
    if (query != null) {
      this.text = query
    } else {
      this.text = ''
    }
    this.service = new ClipService(this.backendHost)
    this.numImages = 20
    this.models = []
    this.images = []
    this.modality = 'image'
    this.blacklist = {}
    this.lastSearch = 'text'
    this.displayCaptions = true
    this.displaySimilarities = false
    this.displayFullCaptions = false
    this.initModels()
  }

  initModels (forceChange) {
    this.service.getIndices().then(l => {
      this.models = l
      if (forceChange || this.currentIndex === '') {
        this.currentIndex = this.models[0]
      }
    })
  }

  static get properties () {
    return {
      service: { type: Object },
      images: { type: Array },
      image: { type: String },
      imageUrl: { type: String },
      text: { type: String },
      numImages: { type: Number },
      modality: { type: String },
      models: { type: Array },
      currentIndex: { type: String },
      backendHost: { type: String },
      blacklist: { type: Object },
      displaySimilarities: { type: Boolean },
      displayCaptions: { type: Boolean },
      displayFullCaptions: { type: Boolean }
    }
  }

  firstUpdated () {
    const searchElem = this.shadowRoot.getElementById('searchBar')
    searchElem.addEventListener('keyup', e => { if (e.keyCode === 13) { this.textSearch() } })
  }

  updated (_changedProperties) {
    if (_changedProperties.has('backendHost')) {
      this.service.backend = this.backendHost
      this.initModels(true)
    }
    if (_changedProperties.has('image')) {
      if (this.image !== undefined) {
        this.imageSearch()
      }
    }
    if (_changedProperties.has('imageUrl')) {
      if (this.imageUrl !== undefined) {
        this.imageUrlSearch()
      }
    }
    if (_changedProperties.has('modality')) {
      if (this.image !== undefined || this.text !== '' || this.imageUrl !== undefined) {
        this.redoSearch()
      }
    }
  }

  async redoSearch () {
    if (this.lastSearch === 'text') {
      this.textSearch()
    } else if (this.lastSearch === 'image') {
      this.imageSearch()
    } else if (this.lastSearch === 'imageUrl') {
      this.imageUrlSearch()
    }
  }

  async textSearch () {
    this.image = undefined
    this.imageUrl = undefined
    const results = await this.service.callClipService(this.text, null, null, this.modality, this.numImages, this.currentIndex)
    console.log(results)
    this.images = results
    this.lastSearch = 'text'
  }

  async imageSearch () {
    this.text = ''
    this.imageUrl = undefined
    const results = await this.service.callClipService(null, this.image, null, this.modality, this.numImages, this.currentIndex)
    console.log(results)
    this.images = results
    this.lastSearch = 'image'
  }

  async imageUrlSearch () {
    this.text = ''
    this.image = undefined
    const results = await this.service.callClipService(null, null, this.imageUrl, this.modality, this.numImages, this.currentIndex)
    console.log(results)
    this.images = results
    this.lastSearch = 'imageUrl'
  }
  static get styles () {
    return css`
    input:-webkit-autofill,
    input:-webkit-autofill:hover,
    input:-webkit-autofill:focus,
    input:-webkit-autofill:active {
        -webkit-transition: "color 9999s ease-out, background-color 9999s ease-out";
        -webkit-transition-delay: 9999s;
    }

    figcaption {
      display: table-caption;
      caption-side: bottom;
      background: #fff;
      padding: 0 0px 0px;
    }

    #searchBar, #searchBar:hover, #searchBar:focus, #searchBar:valid {
      border-radius: 25px;
      border-color: #ddd;
      background-color:white;
      border-width:1px;
      width:60%;
      padding:15px;
      outline: none;
      border-style: solid;
      font-size:16px;
      font-family:arial, sans-serif;
    }
    #searchBar:hover, #searchBar:focus {
      box-shadow: 0px 0px 7px  #ccc;
    }
    #all {
      margin-left:2%;
      margin-right:2%;
      margin-top:2%;
    }
    #inputSearchBar:hover > #searchBar {
      box-shadow: 0px 0px 7px  #ccc !important;
    }
    #imageSearch {
      width: 22px;
      margin-left:0.5%;
      vertical-align:middle;
      cursor:pointer;
    }
    #textSearch {
      width: 22px;
      margin-left:1.5%;
      vertical-align:middle;
      cursor:pointer;
    }
    .subImageSearch {
      width: 22px;
      height: 22px:
      cursor:pointer;
      float:right;
      z-index:90;
      display:None;
    }
    .subTextSearch {
      width: 22px;
      height: 22px:
      cursor:pointer;
      margin-left:5%;
      margin-right:5%;
      float:right;
      z-index:90;
      display:None;
    }
    figure:hover > .subImageSearch {
      display:inline;
      cursor:pointer;
    }
    figure:hover > .subTextSearch {
      display:inline;
      cursor:pointer;
    }
    #products {
      margin-top:50px;
      width:87%;
      float:right;
      display: inline-grid;
    }
    @media (min-width: 400px) {
      #products {
        grid-template-columns: repeat(2, 1fr);
      }
    }
    
    @media (min-width: 700px) {
      #products{
        grid-template-columns: repeat(4, 1fr);
      }
    }
    
    @media (min-width: 1000px) {
      #products {
        grid-template-columns: repeat(5, 1fr);
      }
    }
    
    @media (min-width: 1300px) {
      #products {
        grid-template-columns: repeat(7, 1fr);
      }
    }
    
    @media (min-width: 1600px) {
      #products{
        grid-template-columns: repeat(8, 1fr);
      }
    }
    #filter {
      margin-top:50px;
      width:13%;
      float:left;
    }
    #searchLine {
        
      margin-left:13%;
    }

    `
  }

  updateImage (file) {
    var reader = new FileReader()
    reader.readAsDataURL(file)
    reader.onload = () => {
      this.image = reader.result.split(',')[1]
    }
    reader.onerror = (error) => {
      console.log('Error: ', error)
    }
  }

  renderImage (image) {
    let src
    if (image['image'] !== undefined) {
      src = `data:image/png;base64, ${image['image']}`
    }
    if (image['url'] !== undefined) {
      src = image['url']
    }
    /*
    // useful for testing broken images
    const hashCode = s => s.split('').reduce((a,b)=>{a=((a<<5)-a)+b.charCodeAt(0);return a&a},0)

    const disp =  hashCode(src) % 2 == 0
    src = (disp ? "" : "sss") +src
    */
    return html`
    <figure style="margin:5px;width:150px;display:table" 
    style=${'margin:1px; ' + (this.blacklist[src] !== undefined ? 'display:none' : 'display:inline')}>
     ${this.displaySimilarities ? html`<p>${(image['similarity']).toFixed(4)}</p>` : ``}

     <img src="assets/search.png" class="subTextSearch" @click=${() => { this.text = image['caption']; this.textSearch() }} />
     <img src="assets/image-search.png" class="subImageSearch" @click=${() => {
    if (image['image'] !== undefined) {
      this.image = image['image']
    } else if (image['url'] !== undefined) {
      this.imageUrl = image['url']
    }
  }} />
      <img width="150" src="${src}" alt="${image['caption']}"" title="${image['caption']}"
      @error=${() => { this.blacklist = { ...this.blacklist, ...{ [src]: true } } }} />
      
      ${this.displayCaptions ? html`<figcaption style="font-size:13px;width:150px;">
      ${image['caption'].length > 50 && !this.displayFullCaptions ? image['caption'].substr(0, 50) + '...' : image['caption']}</figcaption>` : ''}
    
    
    </figure>
    `
  }

  render () {
    return html`
    <div id="all">
    <div id="searchLine">
      <span id="inputSearchBar">
        <input id="searchBar" type="text" .value=${this.text} @input=${e => { this.text = e.target.value }}/>
        <img src="assets/search.png" id="textSearch" @click=${() => { this.textSearch() }} />
        <img src="assets/image-search.png" id="imageSearch" @click=${() => { this.shadowRoot.getElementById('filechooser').click() }} />
        <input type="file" id="filechooser" style="position:absolute;top:-100px" @change=${() =>
    this.updateImage(this.shadowRoot.getElementById('filechooser').files[0])}>
        Backend url: <input type="text" value=${this.backendHost} @input=${e => { this.backendHost = e.target.value }}/>
        Index: <select @input=${e => { this.currentIndex = e.target.value }}>${this.models.map(model =>
  html`<option value=${model} ?selected=${model === this.currentIndex}>${model}</option>`)}</select>
      </span>
     
    </div>
    <div id="filter">
      ${this.image !== undefined ? html`<img width="100px" src="data:image/png;base64, ${this.image}"" /><br />` : ``}
      ${this.imageUrl !== undefined ? html`<img width="100px" src="${this.imageUrl}"" /><br />` : ``}
      <a href="https://github.com/rom1504/clip-retrieval">Clip retrieval</a> works by converting the text query to a CLIP embedding
      , then using that embedding to query a knn index of clip image embedddings<br /><br />
      <label>Display captions<input type="checkbox" ?checked="${this.displayCaptions}" @click=${() => { this.displayCaptions = !this.displayCaptions }} /></label><br />
      <label>Display full captions<input type="checkbox" ?checked="${this.displayFullCaptions}" @click=${() => { this.displayFullCaptions = !this.displayFullCaptions }} /></label><br />
      <label>Display similarities<input type="checkbox" ?checked="${this.displaySimilarities}" @click=${() => { this.displaySimilarities = !this.displaySimilarities }} /></label><br />
      <label>Search over <select @input=${e => { this.modality = e.target.value }}>${['image', 'text'].map(modality =>
  html`<option value=${modality} ?selected=${modality === this.modality}>${modality}</option>`)}</select>
        <p>This UI may contain results with nudity and is best used by adults. The images are under their own copyright.</p>
     </div>

    <div id="products">
    ${this.images.map(image => this.renderImage(image))}
    </div>
    </div>
    `
  }
}

customElements.define('clip-front', ClipFront)
