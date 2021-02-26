defmodule NxFizzBuzz do
  defdelegate fizz_buzz(n), to: NxFizzBuzz.Util

  def predict_fizz_buzz(params, n) do
    {w1, _, _, _} = params
    {input_size, _} = Nx.shape(w1)

    batch =
      NxFizzBuzz.Util.to_feature([n], input_size)
      |> Nx.tensor()

    NxFizzBuzz.Model.predict(params, batch)
    |> NxFizzBuzz.Util.decode_label(n)
  end
end
