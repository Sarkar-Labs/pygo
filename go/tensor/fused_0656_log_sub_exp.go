package tensor

import "fmt"

func Fused0656LogSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_exp: nil operand")
	}
	r := x.Sub(y)
	r = r.Log()
	r = r.Exp()
	return r, nil
}
func Fused0656LogSubExpName() string { return "log_sub_exp" }
