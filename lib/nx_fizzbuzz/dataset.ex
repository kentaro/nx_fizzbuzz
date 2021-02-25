defmodule NxFizzBuzz.Dataset do
  def generate_dataset(size) do
    max = :math.pow(2, 32) |> round() |> Kernel.-(1)

    randn =
      1..size
      |> Enum.map(fn _ -> Enum.random(0..max) end)

    labels = randn |> Enum.map(&fizz_buzz/1)

    feature =
      randn
      |> Enum.map(fn n ->
        1..127
        |> Enum.reduce([n], fn m, acc ->
          [rem(n, m) | acc]
        end)
        |> Enum.reverse()
      end)

    {
      Nx.tensor(feature, type: {:u, 32}),
      Nx.tensor(labels, type: {:u, 32})
    }
  end

  defp fizz_buzz(n) do
    cond do
      # FizzBuzz
      rem(n, 15) == 0 -> [0, 0, 0, 1]
      # Buzz
      rem(n, 5) == 0 -> [0, 0, 1, 0]
      # Fizz
      rem(n, 3) == 0 -> [0, 1, 0, 0]
      # number
      true -> [n, 0, 0, 0]
    end
  end
end
