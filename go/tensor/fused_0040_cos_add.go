package tensor

import "fmt"

func Fused0040CosAdd(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_add: nil operand")
	}
	r := x.Add(y)
	r = r.Cos()
	return r, nil
}
func Fused0040CosAddName() string { return "cos_add" }
