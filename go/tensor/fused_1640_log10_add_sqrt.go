package tensor

import "fmt"

func Fused1640Log10AddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_sqrt: nil operand")
	}
	r := x.Add(y)
	r = r.Log10()
	r = r.Sqrt()
	return r, nil
}
func Fused1640Log10AddSqrtName() string { return "log10_add_sqrt" }
