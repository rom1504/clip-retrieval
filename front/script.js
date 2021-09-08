const app = new Vue({
  data() {
    return {
      indicesList: [],
      imageList: [],
      backendURL: "",
      selectedIndex: "",
      searchText: "",
      searchImage: "",
      imageCount: 20
    }
  },
  async mounted() {
    this.backendURL = localStorage.getItem("backend-url") || "https://clip.rom1504.fr";
    this.imageCount = Number(localStorage.getItem("image-count")) || 20;
    this.searchText = localStorage.getItem("search-text") || "cat";

    await this.fetchBackend();
    await this.getImages();
  },
  methods: {
    async fetchBackend() {
      this.indicesList = await fetch(`${this.backendURL}/indices-list`).then(d => d.json());
      this.selectedIndex = localStorage.getItem("selected-index") || this.indicesList[0];
    },
    async getImages() {
      let result = await fetch(`${this.backendURL}/knn-service`, {
        method: "POST",
        headers: {
          "content-type": "application/json"
        },
        body: JSON.stringify(
          {
            "text": this.searchText || null,
            "image": this.searchImage || null,
            "modality": "image",
            "num_images": this.imageCount,
            "indice_name": this.selectedIndex
          })
      }).then(d => d.json());
      this.imageList = result.sort((a, b) => b.similarity * 1000 - a.similarity * 1000);
    },
    onImageClick(item) {
      if (item.caption == this.searchText) return;
      this.searchText = item.caption;
      // this.searchImage = item.image_path;
      this.getImages();
    }
  },
  watch: {
    backendURL(url) {
      if (url.endsWith("/")) this.backendURL = url.slice(0, -1);
      localStorage.setItem("backend-url", this.backendURL);
    },
    imageCount(count) {
      localStorage.setItem("image-count", count);
    },
    selectedIndex(index) {
      localStorage.setItem("selected-index", index);
    },
    searchText(text) {
      localStorage.setItem("search-text", text);
    }
  }
});
app.$mount("#app");