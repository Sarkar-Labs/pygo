package tensor

import "fmt"

func Fused0826CosSubAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub_abs: nil operand")
	}
	r := x.Sub(y)
	r = r.Cos()
	r = r.Abs()
	return r, nil
}
func Fused0826CosSubAbsName() string { return "cos_sub_abs" }
