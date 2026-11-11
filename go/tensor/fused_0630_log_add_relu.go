package tensor

import "fmt"

func Fused0630LogAddRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_add_relu: nil operand")
	}
	r := x.Add(y)
	r = r.Log()
	r = r.ReLU()
	return r, nil
}
func Fused0630LogAddReluName() string { return "log_add_relu" }
