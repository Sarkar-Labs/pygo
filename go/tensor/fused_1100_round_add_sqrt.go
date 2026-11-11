package tensor

import "fmt"

func Fused1100RoundAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_add_sqrt: nil operand")
	}
	r := x.Add(y)
	r = r.Round()
	r = r.Sqrt()
	return r, nil
}
func Fused1100RoundAddSqrtName() string { return "round_add_sqrt" }
