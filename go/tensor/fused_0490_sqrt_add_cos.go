package tensor

import "fmt"

func Fused0490SqrtAddCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_cos: nil operand")
	}
	r := x.Add(y)
	r = r.Sqrt()
	r = r.Cos()
	return r, nil
}
func Fused0490SqrtAddCosName() string { return "sqrt_add_cos" }
