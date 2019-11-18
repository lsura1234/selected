const  tf = require( '@tensorflow/tfjs');
 
// Define a model for linear regression.
// const model = tf.sequential();
// model.add(tf.layers.dense({units: 1, inputShape: [1]}));
 
// // Prepare the model for training: Specify the loss and the optimizer.
// model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
 
// // Generate some synthetic data for training.
// const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
// const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);
 
// // Train the model using the data.
// model.fit(xs, ys).then(() => {
//   // Use the model to do inference on a data point the model hasn't seen before:
//   model.predict(tf.tensor2d([5], [1, 1])).print();
// });

const csvUrl = 'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/boston-housing-train.csv';
 
async function run() {
  // We want to predict the column "medv", which represents a median value of a
  // home (in $1000s), so we mark it as a label.
  const csvDataset = tf.data.csv(
    csvUrl, {
      columnConfigs: {
        medv: {
          isLabel: true
        }
      }
    });
    console.log('aaaa => ',csvDataset )
  // Number of features is the number of column names minus one for the label
  // column.
  const numOfFeatures = (await csvDataset.columnNames()).length - 1;
 
  // Prepare the Dataset for training.
  const flattenedDataset =
    csvDataset
    .map(({xs, ys}) => {
      // Convert rows from object form (keyed by column name) to array form.
      return {xs: Object.values(xs), ys: Object.values(ys)};
    })
    .batch(10);
 
  // Define the model.
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [numOfFeatures],
    units: 1
  }));
  model.compile({
    optimizer: tf.train.sgd(0.000001),
    loss: 'meanSquaredError'
  });
 
  // Fit the model using the prepared Dataset
  return model.fitDataset(flattenedDataset, {
    epochs: 5,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(epoch, logs.loss);
      }
    }
  });
}
 
run().then(() => console.log('Done'));