defmodule NxFizzBuzz.Util do
  def fizz_buzz(n) do
    cond do
      rem(n, 15) == 0 -> "FizzBuzz"
      rem(n, 5) == 0 -> "Buzz"
      rem(n, 3) == 0 -> "Fizz"
      true -> n
    end
  end

  # Conduct a feature engineering because it's not interesting if we use the obvious
  # nature of Fizz Buzz problem. So we do that as if we don't know Fizz Buzz well.
  def to_feature(nums, size) do
    nums
    |> Enum.map(fn n ->
      1..size
      |> Enum.reduce([], fn m, acc ->
        [rem(n, m) | acc]
      end)
    end)
    |> Nx.tensor()
    |> Nx.divide(size)
  end

  def encode_label(n) when is_number(n) do
    [1, 0, 0, 0]
  end

  def encode_label("Fizz") do
    [0, 1, 0, 0]
  end

  def encode_label("Buzz") do
    [0, 0, 1, 0]
  end

  def encode_label("FizzBuzz") do
    [0, 0, 0, 1]
  end

  def decode_label(t, n) do
    case Nx.to_scalar(Nx.argmax(t)) do
      0 -> n
      1 -> "Fizz"
      2 -> "Buzz"
      3 -> "FizzBuzz"
    end
  end
end
