const fs = require("fs");
const Papa = require("papaparse");
const tf = require("@tensorflow/tfjs");
const path = require("path");


// Define file paths
const filePath = path.join(__dirname, "file", "tcs-1april-31may.csv");
const outfilePath = path.join(__dirname, "file", "tcs_result.csv");


// Function to load and parse CSV data
function loadCSV(filePath) {
  const csvFile = fs.readFileSync(filePath, "utf8");
  return new Promise((resolve, reject) => {
    Papa.parse(csvFile, {
      header: true,
      dynamicTyping: true,
      complete: (results) => resolve(results.data),
      error: (error) => reject(error),
    });
  });
}


// Convert 'DD-MMM-YYYY' to 'YYYY-MM-DD'
function formatDateToISO(dateStr) {
  const [day, monthStr, year] = dateStr.split("-");
  const month = new Date(Date.parse(monthStr + " 1, 2020")).getMonth(); // Convert month string to month number
  const date = new Date(year, month, day);


  // Format date as 'YYYY-MM-DD'
  const yearFormatted = date.getFullYear();
  const monthFormatted = String(date.getMonth() + 1).padStart(2, "0");
  const dayFormatted = String(date.getDate()).padStart(2, "0");


  return `${yearFormatted}-${monthFormatted}-${dayFormatted}`;
}


// Prepare data for training
function prepareData(data) {
  // Convert dates to timestamps (numeric values)
  const dates = data.map((row) =>
    new Date(formatDateToISO(row.Date)).getTime()
  );
  const prices = data.map((row) => parseFloat(row["close"].replace(/,/g, "")));


  // Ensure no NaN values in dates and prices
  if (dates.some(isNaN) || prices.some(isNaN)) {
    throw new Error("Data contains NaN values");
  }


  // Normalize data (optional, for better model performance)
  const maxDate = Math.max(...dates);
  const minDate = Math.min(...dates);
  const maxPrice = Math.max(...prices);
  const minPrice = Math.min(...prices);


  const normalize = (value, min, max) => (value - min) / (max - min);


  const xs = tf.tensor2d(
    dates.map((date) => normalize(date, minDate, maxDate)),
    [dates.length, 1]
  );
  const ys = tf.tensor2d(
    prices.map((price) => normalize(price, minPrice, maxPrice)),
    [prices.length, 1]
  );


  return { xs, ys, minDate, maxDate, minPrice, maxPrice };
}


// Train a linear regression model
async function trainModel(xs, ys) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));


  model.compile({
    optimizer: "sgd",
    loss: "meanSquaredError",
  });


  await model.fit(xs, ys, {
    epochs: 100,
    verbose: 1,
  });


  return model;
}


// Predict prices using the trained model
async function predictPrices(
  model,
  dates,
  minDate,
  maxDate,
  minPrice,
  maxPrice
) {
  const normalizedDates = dates.map(
    (date) => (date - minDate) / (maxDate - minDate)
  );
  const xs = tf.tensor2d(normalizedDates, [normalizedDates.length, 1]);
  const predictions = model.predict(xs);


  const predictedArray = predictions.arraySync().map((prediction) => {
    const value = prediction[0] * (maxPrice - minPrice) + minPrice; // Denormalize
    if (isNaN(value)) {
      console.warn("Prediction is NaN");
      return 0;
    }
    return value;
  });


  return predictedArray;
}


// Calculate MSE and RMSE
function calculateErrorMetrics(actualPrices, predictedPrices) {
  const mse =
    actualPrices.reduce((sum, actual, index) => {
      const error = actual - predictedPrices[index];
      return sum + error * error;
    }, 0) / actualPrices.length;


  const rmse = Math.sqrt(mse);


  return { mse, rmse };
}


// Save data to CSV with cleaned headers
function saveToCSV(filePath, data) {
  const cleanedData = data.map((row) => ({
    ...row,
    Date: row.Date.trim(),
    PredictedPrice: row.PredictedPrice.toFixed(2),
    RMSE: row.RMSE.toFixed(2),
    MSE: row.MSE.toFixed(2),
  }));


  const csv = Papa.unparse(cleanedData, {
    header: true,
    columns: ["Date", "PredictedPrice", "RMSE", "MSE"], // Explicitly define headers
  });


  fs.writeFileSync(filePath, csv, "utf8");
}


// Function to generate next 30 weekdays, avoiding weekends
function getNext30Weekdays(startDate) {
  const weekdays = [];
  let currentDate = new Date(startDate);


  while (weekdays.length < 30) {
    const dayOfWeek = currentDate.getDay();
    if (dayOfWeek !== 0 && dayOfWeek !== 6) {
      // Skip Saturdays and Sundays
      weekdays.push(currentDate.toISOString().split("T")[0]); // Format date as 'YYYY-MM-DD'
    }
    currentDate.setDate(currentDate.getDate() + 1);
  }


  return weekdays;
}


// Main function to orchestrate the workflow
async function main() {
  try {
    const inputFilePath = filePath;
    const outputFilePath = outfilePath;


    // Load and prepare the data
    const data = await loadCSV(inputFilePath);
    const { xs, ys, minDate, maxDate, minPrice, maxPrice } = prepareData(data);


    // Train the model
    const model = await trainModel(xs, ys);


    // Predict on training data
    const predictions = await predictPrices(
      model,
      xs.arraySync().map(([date]) => date * (maxDate - minDate) + minDate),
      minDate,
      maxDate,
      minPrice,
      maxPrice
    );


    // Calculate MSE and RMSE
    const actualPrices = ys
      .arraySync()
      .map((price) => price[0] * (maxPrice - minPrice) + minPrice);
    const { mse, rmse } = calculateErrorMetrics(actualPrices, predictions);


    // Get the first date from the data
    const firstDate = new Date(formatDateToISO(data[0].Date));


    // Generate the next 30 weekdays from the first date
    const next30Weekdays = getNext30Weekdays(firstDate);


    // Predict future prices
    const futureDates = next30Weekdays.map((date) => new Date(date).getTime());
    const futurePredictions = await predictPrices(
      model,
      futureDates,
      minDate,
      maxDate,
      minPrice,
      maxPrice
    );


    // Prepare results for the next 30 weekdays including RMSE and MSE
    const results = next30Weekdays.map((date, index) => ({
      Date: date,
      PredictedPrice: futurePredictions[index],
      RMSE: rmse,
      MSE: mse,
    }));


    // Save predictions and metrics to a new CSV file
    saveToCSV(outputFilePath, results);


    console.log("Predictions and metrics saved to", outputFilePath);
  } catch (error) {
    console.error("Error:", error);
  }
}


main();


