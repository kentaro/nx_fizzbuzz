defmodule NxFizzBuzz.Dataset do
  def generate_dataset(dataset_size \\ 10000, feature_size \\ 20) do
    max = :math.pow(2, 32) |> round() |> Kernel.-(1)

    randn =
      1..dataset_size
      |> Enum.map(fn _ -> Enum.random(0..max) end)

    features =
      randn
      |> NxFizzBuzz.Util.to_feature(feature_size)

    labels =
      randn
      |> Enum.map(&NxFizzBuzz.Util.fizz_buzz/1)
      |> Enum.map(&NxFizzBuzz.Util.encode_label/1)
      |> Nx.tensor()

    {features, labels}
  end
end
