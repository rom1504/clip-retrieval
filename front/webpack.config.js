'use strict'

const { resolve, join } = require('path')
const { merge } = require('webpack-merge')
const CopyWebpackPlugin = require('copy-webpack-plugin')
const HtmlWebpackPlugin = require('html-webpack-plugin')
const { CleanWebpackPlugin } = require('clean-webpack-plugin')

const ENV = process.argv.find(arg => arg.includes('production'))
  ? 'production'
  : 'development'
const OUTPUT_PATH = ENV === 'production' ? resolve('build') : resolve('src')
const INDEX_TEMPLATE = resolve('./src/index.html')

const webcomponentsjs = './node_modules/@webcomponents/webcomponentsjs'

const assets = [
  {
    from: resolve('./src/assets'),
    to: join('assets')
  },
  {
    from: resolve('./config.json'),
    to: join('config.json')
  }
]

const polyfills = [
  {
    from: resolve(`${webcomponentsjs}/webcomponents-*.js`),
    to: join(OUTPUT_PATH, 'vendor')
  },
  {
    from: resolve(`${webcomponentsjs}/bundles/*.js`),
    to: join(OUTPUT_PATH, 'vendor', 'bundles')
  },
  {
    from: resolve(`${webcomponentsjs}/custom-elements-es5-adapter.js`),
    to: join(OUTPUT_PATH, 'vendor')
  }
]

const commonConfig = merge([
  {
    entry: './src/clip-front.js',
    output: {
      path: OUTPUT_PATH,
      filename: '[name].[chunkhash:8].js'
    },
    module: {
      rules: [
        {
          test: /\.css$/,
          use: [
            'to-string-loader',
            'css-loader'
          ]
        },
        {
          test: /\.png|\.gif|\.txt$/,
          use: [
            {
              loader: 'file-loader'
            }
          ]
        }
      ]
    },
    resolve: {
      extensions: ['.js', '.jsx']
    }
  }
])

const developmentConfig = merge([
  {
    devtool: 'cheap-module-source-map',
    plugins: [
      new CopyWebpackPlugin({ patterns: [...polyfills, ...assets] }),
      new HtmlWebpackPlugin({
        template: INDEX_TEMPLATE
      })
    ],

    devServer: {
      compress: true,
      port: 3005,
      historyApiFallback: true,
      host: '0.0.0.0'
    }
  }
])

const productionConfig = merge([
  {
    devtool: 'nosources-source-map',
    plugins: [
      new CleanWebpackPlugin(),
      new CopyWebpackPlugin({ patterns: [...polyfills, ...assets] }),
      new HtmlWebpackPlugin({
        template: INDEX_TEMPLATE,
        filename: 'index.html',
        minify: {
          collapseWhitespace: true,
          removeComments: true,
          minifyCSS: true,
          minifyJS: true
        }
      })
    ]
  }
])

module.exports = mode => {
  if (mode.production) {
    return merge(commonConfig, productionConfig, { mode: 'production' })
  }
  const config = merge(commonConfig, developmentConfig, { mode: 'development' })

  return config
}
