package tensor

import "fmt"

func Fused0635LogAddSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_add_sigmoid: nil operand")
	}
	r := x.Add(y)
	r = r.Log()
	r = r.Sigmoid()
	return r, nil
}
func Fused0635LogAddSigmoidName() string { return "log_add_sigmoid" }
