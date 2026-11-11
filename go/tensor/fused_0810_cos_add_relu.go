package tensor

import "fmt"

func Fused0810CosAddRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_add_relu: nil operand")
	}
	r := x.Add(y)
	r = r.Cos()
	r = r.ReLU()
	return r, nil
}
func Fused0810CosAddReluName() string { return "cos_add_relu" }
