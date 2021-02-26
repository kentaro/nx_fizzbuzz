defmodule NxFizzBuzz.Dataset do
  @default_tensor_opts [type: {:u, 32}]
  def generate_dataset(dataset_size \\ 10000, feature_size \\ 20) do
    max = :math.pow(2, 32) |> round() |> Kernel.-(1)

    randn =
      1..dataset_size
      |> Enum.map(fn _ -> Enum.random(0..max) end)
    features =
      randn
      |> NxFizzBuzz.Util.to_feature(feature_size, @default_tensor_opts)
      |> Nx.tensor(@default_tensor_opts)
    labels =
      randn
      |> Enum.map(&fizz_buzz/1)
      |> Enum.map(&NxFizzBuzz.Util.encode_label/1)
      |> Nx.tensor(@default_tensor_opts)

    {features, labels}
  end

  defp fizz_buzz(n) do
    cond do
      rem(n, 15) == 0 -> "FizzBuzz"
      rem(n, 5) == 0 -> "Buzz"
      rem(n, 3) == 0 -> "Fizz"
      true -> n
    end
  end
end
