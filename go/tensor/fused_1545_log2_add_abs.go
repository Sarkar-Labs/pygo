package tensor

import "fmt"

func Fused1545Log2AddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_abs: nil operand")
	}
	r := x.Add(y)
	r = r.Log2()
	r = r.Abs()
	return r, nil
}
func Fused1545Log2AddAbsName() string { return "log2_add_abs" }
