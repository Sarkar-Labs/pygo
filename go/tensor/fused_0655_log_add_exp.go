package tensor

import "fmt"

func Fused0655LogAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_add_exp: nil operand")
	}
	r := x.Add(y)
	r = r.Log()
	r = r.Exp()
	return r, nil
}
func Fused0655LogAddExpName() string { return "log_add_exp" }
