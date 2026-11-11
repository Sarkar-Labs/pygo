package tensor

import "fmt"

func Fused0065GeluAdd(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_add: nil operand")
	}
	r := x.Add(y)
	r = r.GELU()
	return r, nil
}
func Fused0065GeluAddName() string { return "gelu_add" }
