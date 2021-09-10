/* globals customElements */
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
      this.text = 'cat'
    }
    this.service = new ClipService(this.backendHost)
    this.numImages = 20
    this.models = []
    this.images = []
    this.modality = 'image'
    this.blacklist = {}
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
      text: { type: String },
      numImages: { type: Number },
      models: { type: Array },
      currentIndex: { type: String },
      backendHost: { type: String },
      blacklist: { type: Object },
      displaySimilarities: { type: Boolean },
      displayCaptions: { type: Boolean }
    }
  }

  firstUpdated () {
    const searchElem = this.shadowRoot.getElementById('searchBar')
    searchElem.addEventListener('keyup', e => { if (e.keyCode === 13) { this.search() } })
  }

  updated (_changedProperties) {
    if (_changedProperties.has('backendHost')) {
      this.service.backend = this.backendHost
      this.initModels(true)
    }
  }

  async search () {
    const results = await this.service.callClipService(this.text, null, this.modality, this.numImages, this.currentIndex)
    console.log(results)
    this.images = results
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
      <img width="150" src="${src}" alt="${image['caption']}"" title="${image['caption']}"
      @error=${() => { this.blacklist = { ...this.blacklist, ...{ [src]: true } } }} />
      
      ${this.displayCaptions ? html`<figcaption>${image['caption']}</figcaption>` : ''}
    
    
    </figure>
    `
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
    #textSearch {
      width: 22px;
      margin-left:1.5%;
      vertical-align:middle;
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

  render () {
    return html`
    <div id="all">
    <div id="searchLine">
      <span id="inputSearchBar">
        <input id="searchBar" type="text" value=${this.text} @input=${e => { this.text = e.target.value }}/>
        <img src="assets/search.png" id="textSearch" @click=${e => { this.search() }} />
        Backend url: <input type="text" value=${this.backendHost} @input=${e => { this.backendHost = e.target.value }}/>
        Index: <select @input=${e => { this.currentIndex = e.target.value }}>${this.models.map(model =>
  html`<option value=${model} ?selected=${model === this.currentIndex}>${model}</option>`)}</select>
      </span>
     
    </div>
    <div id="filter">
      <a href="https://github.com/rom1504/clip-retrieval">Clip retrieval</a> works by converting the text query to a CLIP embedding
      , then using that embedding to query a knn index of clip image embedddings<br /><br />
      <label>Display captions<input type="checkbox" @click=${() => { this.displayCaptions = !this.displayCaptions }} /></label><br />
      <label>Display similarities<input type="checkbox" @click=${() => { this.displaySimilarities = !this.displaySimilarities }} /></label><br />
      <label>Search over <select @input=${e => { this.modality = e.target.value }}>${['image', 'text'].map(modality =>
  html`<option value=${modality} ?selected=${modality === this.modality}>${modality}</option>`)}</select>
     </div>

    <div id="products">
    ${this.images.map(image => this.renderImage(image))}
    </div>
    </div>
    `
  }
}

customElements.define('clip-front', ClipFront)
