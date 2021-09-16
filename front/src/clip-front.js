/* globals customElements, FileReader */
import { LitElement, html, css } from 'lit-element'
import ClipService from './clip-service'

class ClipFront extends LitElement {
  constructor () {
    super()
    const urlParams = new URLSearchParams(window.location.search)
    const back = urlParams.get('back')
    const index = urlParams.get('index')
    const query = urlParams.get('query')
    const imageUrl = urlParams.get('imageUrl')
    if (index != null) {
      this.currentIndex = index
    } else {
      this.currentIndex = ''
    }
    if (back != null) {
      this.backendHost = back
    } else {
      this.backendHost = 'https://clip-big.rom1504.fr' // put something here
    }
    if (query != null) {
      this.text = query
    } else {
      this.text = ''
    }
    this.service = new ClipService(this.backendHost)
    this.numImages = 20
    this.indices = []
    this.images = []
    this.modality = 'image'
    this.blacklist = {}
    this.lastSearch = 'text'
    this.displayCaptions = true
    this.displaySimilarities = false
    this.displayFullCaptions = false
    this.safeMode = true
    this.firstLoad = true
    this.imageUrl = imageUrl === null ? undefined : imageUrl
    this.initIndices()
  }

  initIndices (forceChange) {
    this.service.getIndices().then(l => {
      this.indices = l
      if (forceChange || this.currentIndex === '') {
        this.currentIndex = this.indices[0]
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
      indices: { type: Array },
      currentIndex: { type: String },
      backendHost: { type: String },
      blacklist: { type: Object },
      displaySimilarities: { type: Boolean },
      displayCaptions: { type: Boolean },
      displayFullCaptions: { type: Boolean },
      safeMode: { type: Boolean }
    }
  }

  firstUpdated () {
    const searchElem = this.shadowRoot.getElementById('searchBar')
    searchElem.addEventListener('keyup', e => { if (e.keyCode === 13) { this.textSearch() } })
  }

  updated (_changedProperties) {
    if (_changedProperties.has('backendHost')) {
      this.service.backend = this.backendHost
      this.initIndices(!this.firstLoad)
      this.firstLoad = false
      this.setUrlParams()
    }
    if (_changedProperties.has('currentIndex')) {
      this.setUrlParams()
    }
    if (_changedProperties.has('image')) {
      if (this.image !== undefined) {
        this.imageSearch()
        return
      }
    }
    if (_changedProperties.has('imageUrl')) {
      if (this.imageUrl !== undefined) {
        this.imageUrlSearch()
        return
      }
    }
    if (_changedProperties.has('modality') || _changedProperties.has('currentIndex')) {
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

  setUrlParams () {
    const urlParams = new URLSearchParams(window.location.search)
    if (this.text !== '') {
      urlParams.set('query', this.text)
    } else {
      urlParams.delete('query')
    }
    if (this.imageUrl !== undefined) {
      urlParams.set('imageUrl', this.imageUrl)
    } else {
      urlParams.delete('imageUrl')
    }
    urlParams.set('back', this.backendHost)
    urlParams.set('index', this.currentIndex)
    window.history.pushState({}, '', '?' + urlParams.toString())
  }

  async textSearch () {
    if (this.text === '') {
      return
    }
    this.image = undefined
    this.imageUrl = undefined
    const results = await this.service.callClipService(this.text, null, null, this.modality, this.numImages, this.currentIndex)
    console.log(results)
    this.images = results
    this.lastSearch = 'text'
    this.setUrlParams()
  }

  async imageSearch () {
    this.text = ''
    this.imageUrl = undefined
    const results = await this.service.callClipService(null, this.image, null, this.modality, this.numImages, this.currentIndex)
    console.log(results)
    this.images = results
    this.lastSearch = 'image'
    this.setUrlParams()
  }

  async imageUrlSearch () {
    this.text = ''
    this.image = undefined
    const results = await this.service.callClipService(null, null, this.imageUrl, this.modality, this.numImages, this.currentIndex)
    console.log(results)
    this.images = results
    this.lastSearch = 'imageUrl'
    this.setUrlParams()
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
      overflow: hidden;
      caption-side: bottom;
      background: #fff;
      padding: 0 0px 0px;
      max-width: 170px;
      word-wrap: break-word;
    }

    #searchBar, #searchBar:hover, #searchBar:focus, #searchBar:valid {
      border-radius: 25px;
      border-color: #ddd;
      background-color:white;
      border-width:1px;
      padding:15px;
      outline: none;
      border-style: solid;
      margin-left:0.5%;
      width: 85%;
    }

    #searchBar:hover, #searchBar:focus {
      box-shadow: 0px 0px 7px  #ccc;
    }
    
    #all {
      margin: 2% auto;
      margin-top:2%;
      font-family: 'Palanquin', sans-serif;
    }
    #inputSearchBar:hover > #searchBar {
      box-shadow: 0px 0px 7px  #ccc !important;
    }
    #imageSearch {
      width: 22px;
      margin-left:5%;
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
      width:100%;
    }

    figure,img.pic,figcaption {
      width:95%;
      padding:2.5%;
    }

    #holder {
      margin 0 auto;
      display: inline-grid;
    }

    figure p {
      font-weight:bold;
    }

    #filter {
      padding: 10px;
      width: 15%;
      min-width: 200px;
      max-width: 400px;
      float: right;
    }
    #main {
      padding:10px;
      width: 75%;
      margin:0 auto;
      float:right;
    }

    .section {
      margin-top:20px;
      float:left;
    }

    .one_of_four {
      width: 25%;
      min-width:200px;
    }

    .two_of_four {
      width: 50%;
      min-width:200px;
    }

    .queryimg, .querytxt {
      float:left;
      display: flex;
      align-items: center;
    }

    .querytxt {
      width: 85%;
    }

    #searchLine {
      display: flex;
      align-items: center;
      clear: both;
    }

    h4 {
      margin-top: 0;
    }

    @media (min-width: 600px) {
      #holder {
        grid-template-columns: repeat(2, 1fr);
      }
      #main {
        width: 60%;
      }
    }

    @media (min-width: 768px) {
      #holder{
        grid-template-columns: repeat(3, 1fr);
      }
      #main {
        width: 67%;
      }
    }
    
    @media (min-width: 910px) {
      #holder{
        grid-template-columns: repeat(4, 1fr);
      }
      #main {
        width: 73%;
      }
    }
    
    @media (min-width: 1100px) {
      #holder {
        grid-template-columns: repeat(5, 1fr);
      }
      #main {
        width: 78%;
      }
    }

    @media (min-width: 1280px) {
      #holder {
        grid-template-columns: repeat(6, 1fr);
      }
    }
    
    @media (min-width: 1430px) {
      #holder {
        grid-template-columns: repeat(7, 1fr);
      }
    }
    
    @media (min-width: 1780px) {
      #holder{
        grid-template-columns: repeat(8, 1fr);
      }
      #filter {
        width:18%;
        max-width:400px;
      }
      .one_of_four {
        width: 50%;
      }
    
      .two_of_four {
        width: 100%;
      }
    }
    
    @media screen and (max-width: 600px) {
      #filter { 
       float: left;
       min-width:100px;
       margin:0 auto;
     }
     .one_of_four {
      width: 50%;
    }

    .two_of_four {
      width: 100%;
    }
      #main {
        float:right;
        width:45%;
     }
   }
 }
    `
  }

  isSafe (image) {
    if ((image['NSFW'] === 'UNSURE' || image['NSFW'] === 'NSFW')) {
      return false
    }
    const badWords = ['boob', 'sexy', 'ass', 'hot', 'mature', 'nude', 'naked', 'porn', 'xvideo',
      'ghetto', 'tube', 'hump', 'fuck', 'dick', 'whore', 'masturbate', 'video', 'puss', 'erotic']
    if (badWords.some(word => (image['url'] !== undefined && image['url'].toLowerCase().includes(word)) ||
    (image['caption'] !== undefined && image['caption'].toLowerCase().includes(word)))) {
      return false
    }
    return true
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
    <figure style="margin:5px;display:table" 
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
      <img class="pic" src="${src}" alt="${image['caption']}"" title="${image['caption']}"
      @error=${() => { this.blacklist = { ...this.blacklist, ...{ [src]: true } } }} />
      
      ${this.displayCaptions ? html`<figcaption>
      ${image['caption'].length > 50 && !this.displayFullCaptions ? image['caption'].substr(0, 50) + '...' : image['caption']}</figcaption>` : ''}
    
    
    </figure>
    `
  }

  render () {
    const filteredImages = this.images.filter(image => !this.safeMode || this.isSafe(image))

    return html`
    <div id="all">
      <div id= "main">
        <div id="searchLine">
          <div id="inputSearchBar" class="querytxt">
            <input id="searchBar" type="text" .value=${this.text} @input=${e => { this.text = e.target.value }}/>
            <img src="assets/search.png" id="textSearch" @click=${() => { this.textSearch() }} />
          </div><div class="queryimg">
            ${this.image !== undefined ? html`<img width="100px" src="data:image/png;base64, ${this.image}"" />` : ``}
            ${this.imageUrl !== undefined ? html`<img width="100px" src="${this.imageUrl}"" />` : ``}
            <img src="assets/image-search.png" id="imageSearch" @click=${() => { this.shadowRoot.getElementById('filechooser').click() }} />
            <input type="file" id="filechooser" style="position:absolute;top:-100px" @change=${() =>
    this.updateImage(this.shadowRoot.getElementById('filechooser').files[0])}>
          </div>
        </div>
        <div id="products">
          <div id="holder">
          ${filteredImages.map(image => this.renderImage(image))}
          ${this.safeMode && this.images.length !== 0 && filteredImages.length === 0 ? 'Displaying only nice pictures in safe mode!' : ''}
          </div>
        </div>
      </div>
      <div id="filter">
        <div class="section one_of_four">
          <h4>Backend controls</h4>
          Insert URL: <br /><input type="text" value=${this.backendHost} @input=${e => { this.backendHost = e.target.value }}/><br />
          Select Index: <br /><select @input=${e => { this.currentIndex = e.target.value }}>${this.indices.map(index =>
  html`<option value=${index} ?selected=${index === this.currentIndex}>${index}</option>`)}</select><br />
        </div>
        <div class="section one_of_four"> 
          <h4>Display controls</h4>
          <label>Display captions<input type="checkbox" ?checked="${this.displayCaptions}" @click=${() => { this.displayCaptions = !this.displayCaptions }} /></label><br />
          <label>Display full captions<input type="checkbox" ?checked="${this.displayFullCaptions}" @click=${() => { this.displayFullCaptions = !this.displayFullCaptions }} /></label><br />
          <label>Display similarities<input type="checkbox" ?checked="${this.displaySimilarities}" @click=${() => { this.displaySimilarities = !this.displaySimilarities }} /></label><br />
          <label>Safe mode<input type="checkbox" ?checked="${this.safeMode}" @click=${() => { this.safeMode = !this.safeMode }} /></label><br />
          <label>Search over <select @input=${e => { this.modality = e.target.value }}>${['image', 'text'].map(modality =>
  html`<option value=${modality} ?selected=${modality === this.modality}>${modality}</option>`)}</select>
        </div>
        <div class="section two_of_four">
          <h4>Info</h4>
          <p><a href="https://github.com/rom1504/clip-retrieval">Clip retrieval</a> works by converting the text query to a CLIP embedding
          , then using that embedding to query a knn index of clip image embedddings</p>
          <p>This UI may contain results with nudity and is best used by adults. The images are under their own copyright.</p>
          <p>Are you seeing near duplicates ? KNN search are good at spotting those, especially so in large datasets.</p>
        </div>
      </div>
    </div>
    `
  }
}

customElements.define('clip-front', ClipFront)
