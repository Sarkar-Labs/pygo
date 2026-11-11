package tensor

import "fmt"

func Fused1556Log2SubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_exp: nil operand")
	}
	r := x.Sub(y)
	r = r.Log2()
	r = r.Exp()
	return r, nil
}
func Fused1556Log2SubExpName() string { return "log2_sub_exp" }
