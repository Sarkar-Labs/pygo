package tensor

import "fmt"

func Fused0491SqrtSubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_cos: nil operand")
	}
	r := x.Sub(y)
	r = r.Sqrt()
	r = r.Cos()
	return r, nil
}
func Fused0491SqrtSubCosName() string { return "sqrt_sub_cos" }
