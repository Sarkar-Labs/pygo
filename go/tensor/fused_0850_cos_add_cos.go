package tensor

import "fmt"

func Fused0850CosAddCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_add_cos: nil operand")
	}
	r := x.Add(y)
	r = r.Cos()
	r = r.Cos()
	return r, nil
}
func Fused0850CosAddCosName() string { return "cos_add_cos" }
