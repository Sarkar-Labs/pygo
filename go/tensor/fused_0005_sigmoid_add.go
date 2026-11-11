package tensor

import "fmt"

func Fused0005SigmoidAdd(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add: nil operand")
	}
	r := x.Add(y)
	r = r.Sigmoid()
	return r, nil
}
func Fused0005SigmoidAddName() string { return "sigmoid_add" }
