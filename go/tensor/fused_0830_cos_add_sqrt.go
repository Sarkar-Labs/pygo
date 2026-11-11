package tensor

import "fmt"

func Fused0830CosAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_add_sqrt: nil operand")
	}
	r := x.Add(y)
	r = r.Cos()
	r = r.Sqrt()
	return r, nil
}
func Fused0830CosAddSqrtName() string { return "cos_add_sqrt" }
