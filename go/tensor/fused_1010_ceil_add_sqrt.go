package tensor

import "fmt"

func Fused1010CeilAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_sqrt: nil operand")
	}
	r := x.Add(y)
	r = r.Ceil()
	r = r.Sqrt()
	return r, nil
}
func Fused1010CeilAddSqrtName() string { return "ceil_add_sqrt" }
