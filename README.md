# NxFizzBuzz

`FizzBuzz` problem got solved by deep learning!

This repository shows an inductive programming demo using [Nx](https://github.com/elixir-nx/nx), a multi-dimensional tensors library for Elixir.

## Usage

```elixir
# generate a dataset
{features, labels} = NxFizzBuzz.Dataset.generate_dataset(10000, 20)

# train the model by the dataset
params = NxFizzBuzz.Model.fit(features, labels, epoch: 100, batch_size: 50, hidden_size: 8)

# predict answers
1..100
|> Enum.each(fn n ->
  NxFizzBuzz.predict_fizz_buzz(params, n)
  |> IO.puts()
end)
```

## Evaluation

It took about 500 secs to train the model using 10,000 data in 100 times epochs. Then 100% of accuracy got achieved.

For the experiment, I used MacBook Pro (2018) with 2.7 GHz Quad Cores Intel Core i7. EXLA was not enabled and no GPUs were used.

```
$ time mix run examples/fizz_buzz.exs

(snip)

97: prediction: 97, answer: 97, matched?: true
98: prediction: 98, answer: 98, matched?: true
99: prediction: Fizz, answer: Fizz, matched?: true
100: prediction: Buzz, answer: Buzz, matched?: true
================
Accuracy: 1.0

________________________________________________________
Executed in  487.17 secs   fish           external
   usr time  477.32 secs  126.00 micros  477.32 secs
   sys time   48.84 secs  688.00 micros   48.84 secs
```

The result shown below is quite obvious because the dataset genearted by genuin `FizzBuzz` function are totally explanable deterministically. You have to notice it's a toy trial for an Nx demo ;)

## Acknowledgement

This code is heavily inspired by the article below:

[機械学習でFizzBuzzを実現する](https://zenn.dev/tokoroten/articles/c311cf6e3fc8ac)

## Author

Kentaro Kuribayashi &lt;kentarok@gmail.com&gt;
