defmodule NxFizzBuzz.Util do
  def to_feature(nums, size, opts \\ []) do
    opts = Keyword.put_new(opts, :type, {:u, 32})
    nums
    |> Enum.map(fn n ->
      1..size
      |> Enum.reduce([], fn m, acc ->
        [rem(n, m) | acc]
      end)
    end)
    |> Nx.tensor(opts)
  end

  def encode_label(n) when is_number(n) do
    [n, 0, 0, 0]
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
end
