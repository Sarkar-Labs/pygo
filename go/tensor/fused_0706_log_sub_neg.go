package tensor

import "fmt"

func Fused0706LogSubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_neg: nil operand")
	}
	r := x.Sub(y)
	r = r.Log()
	r = r.Neg()
	return r, nil
}
func Fused0706LogSubNegName() string { return "log_sub_neg" }
