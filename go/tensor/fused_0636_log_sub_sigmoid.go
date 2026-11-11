package tensor

import "fmt"

func Fused0636LogSubSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_sigmoid: nil operand")
	}
	r := x.Sub(y)
	r = r.Log()
	r = r.Sigmoid()
	return r, nil
}
func Fused0636LogSubSigmoidName() string { return "log_sub_sigmoid" }
