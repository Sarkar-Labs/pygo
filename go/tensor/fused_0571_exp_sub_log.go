package tensor

import "fmt"

func Fused0571ExpSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_log: nil operand")
	}
	r := x.Sub(y)
	r = r.Exp()
	r = r.Log()
	return r, nil
}
func Fused0571ExpSubLogName() string { return "exp_sub_log" }
