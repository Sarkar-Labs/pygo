package tensor

import "fmt"

func Fused1555Log2AddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_exp: nil operand")
	}
	r := x.Add(y)
	r = r.Log2()
	r = r.Exp()
	return r, nil
}
func Fused1555Log2AddExpName() string { return "log2_add_exp" }
