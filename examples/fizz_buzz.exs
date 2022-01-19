# generate a dataset
{features, labels} = NxFizzBuzz.Dataset.generate_dataset(10000, 20)

# train the model by the dataset
params = NxFizzBuzz.Model.fit(features, labels, epoch: 50, batch_size: 50, hidden_size: 8)

# predict answers
max = 100
match_count =
  1..max
  |> Enum.count(fn n ->
    pred = NxFizzBuzz.predict_fizz_buzz(params, n)
    answer = NxFizzBuzz.fizz_buzz(n)
    matched = pred == answer
    IO.puts("#{n}: prediction: #{pred}, answer: #{answer}, matched?: #{matched}")
    matched
end)

# result
IO.puts("================")
IO.puts("Accuracy: #{match_count / max}")
