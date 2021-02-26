defmodule NxFizzBuzz.Model do
  import Nx.Defn

  @default_epoch 10
  @default_batch_size 50
  @default_learning_rate 0.01
  @default_hidden_size 64

  def fit(x, y, opts \\ []) do
    opts =
      opts
      |> Keyword.put_new(:epoch, @default_epoch)
      |> Keyword.put_new(:batch_size, @default_batch_size)
      |> Keyword.put_new(:learning_rate, @default_learning_rate)
      |> Keyword.put_new(:hidden_size, @default_hidden_size)

    {_, input_size} = Nx.shape(x)

    init_params = {
      Nx.random_uniform({input_size, opts[:hidden_size]}, 0.0, 1.0, names: [:input, :hidden]),
      Nx.random_uniform({opts[:hidden_size]}, 0.0, 1.0, names: [:hidden]),
      Nx.random_uniform({opts[:hidden_size], 4}, 0.0, 1.0, names: [:hidden, :output]),
      Nx.random_uniform({4}, 0.0, 1.0, names: [:output])
    }

    zip =
      Enum.zip(
        Nx.to_batched_list(x, opts[:batch_size]),
        Nx.to_batched_list(y, opts[:batch_size])
      )
      |> Enum.with_index()

    for e <- 1..opts[:epoch],
        {{x_batch, y_batch}, b} <- zip,
        reduce: init_params do
      params ->
        IO.puts("epoch #{e}, batch #{b}")
        update(params, x_batch, y_batch, opts)
    end
  end

  defn predict({w1, b1, w2, b2}, batch) do
    batch
    |> Nx.dot(w1)
    |> Nx.add(b1)
    |> Nx.logistic()
    |> Nx.dot(w2)
    |> Nx.add(b2)
    |> softmax()
  end

  defnp update({w1, b1, w2, b2} = params, x, y, opts \\ []) do
    {grad_w1, grad_b1, grad_w2, grad_b2} = grad(params, loss(params, x, y))

    {
      w1 - grad_w1 * opts[:learning_rate],
      b1 - grad_b1 * opts[:learning_rate],
      w2 - grad_w2 * opts[:learning_rate],
      b2 - grad_b2 * opts[:learning_rate]
    }
  end

  defnp loss({w1, b1, w2, b2}, x, y) do
    preds = predict({w1, b1, w2, b2}, x)
    -Nx.sum(Nx.mean(Nx.log(preds) * y, axes: [:output]))
  end

  defnp softmax(t) do
    Nx.exp(t) / Nx.sum(Nx.exp(t), axes: [:output], keep_axes: true)
  end
end
