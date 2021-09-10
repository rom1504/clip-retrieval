#!/usr/bin/env node

const express = require('express')
const compression = require('compression')
const path = require('path')

// Create our app
const app = express()

app.use(compression())
app.use(express.static(path.join(__dirname, './build')))

// Start the server
const server = app.listen(process.argv[2] === undefined ? 8080 : process.argv[2], function () {
  console.log('Server listening on port ' + server.address().port)
})
