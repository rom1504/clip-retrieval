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
      blacklist: { type: Object }
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
    const results = await this.service.callClipService(this.text, null, 'image', this.numImages, this.currentIndex)
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
    return html`<img width="150" src="${src}" alt="${image['caption']}"" title="${image['caption']}"
    style=${'margin:1px; ' + (this.blacklist[src] !== undefined ? 'display:none' : 'display:inline')}
    @error=${() => { this.blacklist = { ...this.blacklist, ...{ [src]: true } } }} />`
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
      <a href="https://github.com/rom1504/clip-retrieval">Clip retrieval</a> works by using embeddings computed with the CLIP model in an efficient knn index.<br />
    </div>

    <div id="products">
    ${this.images.map(image => this.renderImage(image))}
    </div>
    </div>
    `
  }
}

customElements.define('clip-front', ClipFront)
