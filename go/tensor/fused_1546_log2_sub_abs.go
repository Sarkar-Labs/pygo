package tensor

import "fmt"

func Fused1546Log2SubAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_abs: nil operand")
	}
	r := x.Sub(y)
	r = r.Log2()
	r = r.Abs()
	return r, nil
}
func Fused1546Log2SubAbsName() string { return "log2_sub_abs" }
