package tensor

import "fmt"

func Fused0560ExpAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_sqrt: nil operand")
	}
	r := x.Add(y)
	r = r.Exp()
	r = r.Sqrt()
	return r, nil
}
func Fused0560ExpAddSqrtName() string { return "exp_add_sqrt" }
