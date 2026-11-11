package tensor

import "fmt"

func Fused0401AbsSubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_cos: nil operand")
	}
	r := x.Sub(y)
	r = r.Abs()
	r = r.Cos()
	return r, nil
}
func Fused0401AbsSubCosName() string { return "abs_sub_cos" }
