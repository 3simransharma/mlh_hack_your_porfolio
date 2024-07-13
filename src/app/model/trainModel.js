const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');


async function loadData() {
  const dataPath = path.join(__dirname, 'public', 'dataset.json');
  const jsonData = await fs.readFile(dataPath, 'utf8');
  return JSON.parse(jsonData);
}

const data = await loadData();


// Tokenize text data
const tokenizeText = (text, vocab) => {
  return text.split(' ').map(word => vocab.indexOf(word) + 1 || 0);
};

const createVocab = (data) => {
  const vocab = [];
  data.forEach(item => {
    item.ingredient.split(' ').forEach(word => {
      if (!vocab.includes(word)) vocab.push(word);
    });
    item.recipe.split(' ').forEach(word => {
      if (!vocab.includes(word)) vocab.push(word);
    });
    item.substitute.split(' ').forEach(word => {
      if (!vocab.includes(word)) vocab.push(word);
    });
  });
  return vocab;
};

const vocab = createVocab(data);

// Prepare input and output tensors
const prepareData = (data, vocab) => {
  const ingredients = data.map(item => tokenizeText(item.ingredient, vocab));
  const recipes = data.map(item => tokenizeText(item.recipe, vocab));
  const substitutes = data.map(item => tokenizeText(item.substitute, vocab));

  const maxLen = Math.max(
    ...ingredients.map(arr => arr.length),
    ...recipes.map(arr => arr.length),
    ...substitutes.map(arr => arr.length)
  );

  const padSequence = (seq, maxLen) => {
    const padded = new Array(maxLen).fill(0);
    seq.forEach((val, idx) => {
      padded[idx] = val;
    });
    return padded;
  };

  const X = ingredients.map((ing, idx) => padSequence(ing.concat(recipes[idx]), maxLen * 2));
  const y = substitutes.map(sub => padSequence(sub, maxLen));

  return {
    inputs: tf.tensor2d(X, [X.length, maxLen * 2]),
    labels: tf.tensor2d(y, [y.length, maxLen])
  };
};

const { inputs, labels } = prepareData(data, vocab);

//here, I'm creating the Neural network model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 256, activation: 'relu', inputShape: [inputs.shape[1]] }));
model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
model.add(tf.layers.dense({ units: labels.shape[1], activation: 'softmax' }));

model.compile({
  optimizer: 'adam',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
});

// Train the model
model.fit(inputs, labels, {
  epochs: 50,
  batchSize: 2,
  validationSplit: 0.2
}).then(() => {
  // Save the model
  model.save('file://./models/model');
});
