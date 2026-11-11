package tensor

import "fmt"

func Fused0705LogAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_add_neg: nil operand")
	}
	r := x.Add(y)
	r = r.Log()
	r = r.Neg()
	return r, nil
}
func Fused0705LogAddNegName() string { return "log_add_neg" }
