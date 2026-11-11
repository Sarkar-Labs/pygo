package tensor

import "fmt"

func Fused0400AbsAddCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_add_cos: nil operand")
	}
	r := x.Add(y)
	r = r.Abs()
	r = r.Cos()
	return r, nil
}
func Fused0400AbsAddCosName() string { return "abs_add_cos" }
